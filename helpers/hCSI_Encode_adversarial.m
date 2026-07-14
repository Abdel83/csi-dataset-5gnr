function csiReport = hCSI_Encode_adversarial(carrier, csirs, H, nVar, alg)
%hCSIEncode_adversarial  CSI encoding com ataque Adversarial Mismatched CSI.


    % Verificar se há ataque
    hasAttack = isfield(alg, 'AdversarialAttack') && ...
                isfield(alg.AdversarialAttack, 'type') && ...
                ~strcmp(alg.AdversarialAttack.type, 'none') && ...
                isfield(alg.AdversarialAttack, 'epsilon') && ...
                alg.AdversarialAttack.epsilon > 0;

    if ~hasAttack
        csiReport = hCSIEncode(carrier, csirs, H, nVar, alg);
        return;
    end

    % Extrair parâmetros
    atkType = lower(alg.AdversarialAttack.type);
    eps     = alg.AdversarialAttack.epsilon;

    % H: [nSub x nSym x nRx x nTx]
    [nSub, nSym, nRx, nTx] = size(H);

    % ================================================================
    %  APLICAR ATAQUE NO DOMÍNIO DO CANAL OFDM
    % ================================================================
    H_attacked = H;

    switch atkType
        case 'rotation'
            % ========================================================
            %  ROTAÇÃO DO SUBESPAÇO DO CANAL
            % ========================================================
            %  Para cada subportadora e símbolo, rotaciona o vetor de
            %  canal no espaço das antenas Tx por um ângulo theta
            
            % ========================================================
            
            theta = eps * pi;  % Ângulo de rotação [0, pi]
            
            % Gerar matriz de rotação aleatória, FIXA por slot
            rng_state = rng;
            rng(carrier.NSlot + 12345);  % Seed determinística
            
            % Criar rotação no espaço nTx-dimensional via Householder
            % Aplicar a mesma rotação para todas as subportadoras
            
            v = randn(nTx, 1) + 1i*randn(nTx, 1);
            v = v / norm(v);
            
            % Matriz de rotação: R = I*cos(theta) + (2*v*v' - I)*sin(theta)
            % rotação por theta no plano definido por v
            I_nTx = eye(nTx);
            R = I_nTx * cos(theta) + (2*(v*v') - I_nTx) * sin(theta);
            
            rng(rng_state);  % Restaurar estado do RNG
            
            for sub = 1:nSub
                for sym = 1:nSym
                    % H(sub,sym,:,:) → [nRx x nTx]
                    Hsub = squeeze(H(sub, sym, :, :));  % [nRx x nTx]
                    
                    % Rotacionar as colunas (antenas Tx)
                    H_attacked(sub, sym, :, :) = Hsub * R;
                end
            end

        
            
            % Fase 2: Rotação
            eps_rot = eps * 0.8;
            theta = eps_rot * pi;
            
            rng_state = rng;
            rng(carrier.NSlot + 12345);
            v = randn(nTx, 1) + 1i*randn(nTx, 1);
            v = v / norm(v);
            I_nTx = eye(nTx);
            R = I_nTx * cos(theta) + (2*(v*v') - I_nTx) * sin(theta);
            rng(rng_state);
            
            for sub = 1:nSub
                for sym = 1:nSym
                    Hsub = squeeze(H_attacked(sub, sym, :, :));
                    H_attacked(sub, sym, :, :) = Hsub * R;
                end
            end

        otherwise
            % Sem ataque
    end

    % ================================================================
    %  CHAMAR hCSIEncode COM O CANAL MANIPULADO
    % ================================================================
    
    
    alg_clean = alg;
    alg_clean.AdversarialAttack = struct('type','none','epsilon',0);
    
    csiReport = hCSIEncode(carrier, csirs, H_attacked, nVar, alg_clean);
end